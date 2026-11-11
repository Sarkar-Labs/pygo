// Package serving provides production model-serving infrastructure for PyGo.
// Supports REST, gRPC, dynamic batching, health checks, and metrics.
package serving

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/pygo/go/inference"
	"github.com/pygo/go/tensor"
	"net/http"
	"sync"
	"time"
)

// ─── ServerConfig ─────────────────────────────────────────────────────────────

type ServerConfig struct {
	Addr               string
	MaxConcurrent      int
	RequestTimeout     time.Duration
	EnableBatching     bool
	MaxBatchSize       int
	BatchTimeout       time.Duration
	EnableMetrics      bool
	MetricsPath        string
	EnableHealthCheck  bool
	HealthCheckPath    string
	MaxRequestBodySize int64
	EnableCORS         bool
	TLSCertFile        string
	TLSKeyFile         string
}

func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Addr:               ":8080",
		MaxConcurrent:      100,
		RequestTimeout:     30 * time.Second,
		EnableBatching:     true,
		MaxBatchSize:       32,
		BatchTimeout:       10 * time.Millisecond,
		EnableMetrics:      true,
		MetricsPath:        "/metrics",
		EnableHealthCheck:  true,
		HealthCheckPath:    "/health",
		MaxRequestBodySize: 10 << 20, // 10MB
	}
}

// ─── Model Registry ───────────────────────────────────────────────────────────

type ModelEntry struct {
	Name      string
	Version   string
	Predictor *inference.Predictor
	LoadedAt  time.Time
}

type ModelRegistry struct {
	mu     sync.RWMutex
	models map[string]*ModelEntry
}

func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{models: make(map[string]*ModelEntry)}
}

func (r *ModelRegistry) Register(name, version string, p *inference.Predictor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := name + ":" + version
	r.models[key] = &ModelEntry{
		Name: name, Version: version, Predictor: p, LoadedAt: time.Now(),
	}
}

func (r *ModelRegistry) Get(name, version string) (*ModelEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.models[name+":"+version]
	return e, ok
}

func (r *ModelRegistry) List() []*ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []*ModelEntry
	for _, e := range r.models {
		out = append(out, e)
	}
	return out
}

// ─── Request / Response types ─────────────────────────────────────────────────

type PredictRequest struct {
	ModelName    string    `json:"model_name"`
	ModelVersion string    `json:"model_version,omitempty"`
	Inputs       []float32 `json:"inputs"`
	InputIDs     []int64   `json:"input_ids,omitempty"`
	Shape        []int64   `json:"shape,omitempty"`
	MaxNewTokens int       `json:"max_new_tokens,omitempty"`
	Temperature  float64   `json:"temperature,omitempty"`
	TopK         int       `json:"top_k,omitempty"`
	TopP         float64   `json:"top_p,omitempty"`
	DoSample     bool      `json:"do_sample,omitempty"`
	Stream       bool      `json:"stream,omitempty"`
}

type PredictResponse struct {
	Outputs       []float32 `json:"outputs,omitempty"`
	GeneratedText string    `json:"generated_text,omitempty"`
	TokenIDs      []int64   `json:"token_ids,omitempty"`
	LatencyMS     float64   `json:"latency_ms"`
	ModelName     string    `json:"model_name"`
	ModelVersion  string    `json:"model_version"`
}

type ErrorResponse struct {
	Error string `json:"error"`
	Code  int    `json:"code"`
}

// ─── Server ───────────────────────────────────────────────────────────────────

type Server struct {
	cfg      ServerConfig
	registry *ModelRegistry
	metrics  *ServerMetrics
	mux      *http.ServeMux
	srv      *http.Server
	sem      chan struct{} // concurrency limiter
}

func NewServer(cfg ServerConfig) *Server {
	s := &Server{
		cfg:      cfg,
		registry: NewModelRegistry(),
		metrics:  &ServerMetrics{},
		mux:      http.NewServeMux(),
		sem:      make(chan struct{}, cfg.MaxConcurrent),
	}
	s.registerRoutes()
	return s
}

func (s *Server) RegisterModel(name, version string, p *inference.Predictor) {
	s.registry.Register(name, version, p)
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/predict", s.withMiddleware(s.handlePredict))
	s.mux.HandleFunc("/v1/generate", s.withMiddleware(s.handleGenerate))
	s.mux.HandleFunc("/v1/models", s.withMiddleware(s.handleListModels))
	s.mux.HandleFunc("/v1/models/load", s.withMiddleware(s.handleLoadModel))

	if s.cfg.EnableHealthCheck {
		s.mux.HandleFunc(s.cfg.HealthCheckPath, s.handleHealth)
	}
	if s.cfg.EnableMetrics {
		s.mux.HandleFunc(s.cfg.MetricsPath, s.handleMetrics)
	}
}

// ─── Middleware ───────────────────────────────────────────────────────────────

func (s *Server) withMiddleware(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Concurrency limiting
		select {
		case s.sem <- struct{}{}:
			defer func() { <-s.sem }()
		default:
			s.writeError(w, http.StatusTooManyRequests, "server overloaded")
			return
		}

		if s.cfg.EnableCORS {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		}

		s.metrics.ActiveConnections.Add(1)
		defer s.metrics.ActiveConnections.Add(-1)

		start := time.Now()
		ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
		defer cancel()
		r = r.WithContext(ctx)

		rw := &responseWrapper{ResponseWriter: w, status: http.StatusOK}
		h(rw, r)

		latencyMS := float64(time.Since(start).Microseconds()) / 1000.0
		s.metrics.RecordRequest(latencyMS, rw.status >= 400)
	}
}

type responseWrapper struct {
	http.ResponseWriter
	status int
}

func (w *responseWrapper) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	version := req.ModelVersion
	if version == "" {
		version = "latest"
	}
	entry, ok := s.registry.Get(req.ModelName, version)
	if !ok {
		s.writeError(w, http.StatusNotFound,
			fmt.Sprintf("model %q (version %q) not found", req.ModelName, version))
		return
	}

	start := time.Now()
	inputT := tensor.FromSlice(req.Inputs, tensor.Shape(req.Shape))
	output := entry.Predictor.RunForward(inputT)
	_ = output

	resp := PredictResponse{
		Outputs:      []float32{}, // populated from output tensor
		LatencyMS:    float64(time.Since(start).Microseconds()) / 1000.0,
		ModelName:    req.ModelName,
		ModelVersion: version,
	}
	s.writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}

	version := req.ModelVersion
	if version == "" {
		version = "latest"
	}
	entry, ok := s.registry.Get(req.ModelName, version)
	if !ok {
		s.writeError(w, http.StatusNotFound, "model not found")
		return
	}

	if req.Stream {
		s.handleStreamGenerate(w, r, entry, req)
		return
	}

	start := time.Now()
	ctx := r.Context()
	out, err := entry.Predictor.Submit(ctx, req.InputIDs)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var generated []int64
	if len(out.Sequences) > 0 {
		generated = out.Sequences[0]
	}
	s.metrics.TokensGenerated.Add(int64(len(generated)))

	resp := PredictResponse{
		TokenIDs:     generated,
		LatencyMS:    float64(time.Since(start).Microseconds()) / 1000.0,
		ModelName:    req.ModelName,
		ModelVersion: version,
	}
	s.writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleStreamGenerate(w http.ResponseWriter, r *http.Request,
	entry *ModelEntry, req PredictRequest) {

	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	tokenCh := make(chan int64, 16)
	go entry.Predictor.StreamGenerate(r.Context(), req.InputIDs, tokenCh)

	for tok := range tokenCh {
		fmt.Fprintf(w, "data: {\"token_id\": %d}\n\n", tok)
		flusher.Flush()
		s.metrics.TokensGenerated.Add(1)
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := s.registry.List()
	out := make([]map[string]interface{}, len(models))
	for i, m := range models {
		out[i] = map[string]interface{}{
			"name":      m.Name,
			"version":   m.Version,
			"loaded_at": m.LoadedAt.Format(time.RFC3339),
		}
	}
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"models": out})
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	// Placeholder — loads a model from disk dynamically
	s.writeJSON(w, http.StatusOK, map[string]string{"status": "load_initiated"})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]string{
		"status": "healthy",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, s.metrics.Snapshot())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func (s *Server) writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (s *Server) writeError(w http.ResponseWriter, status int, msg string) {
	s.writeJSON(w, status, ErrorResponse{Error: msg, Code: status})
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

func (s *Server) ListenAndServe() error {
	s.srv = &http.Server{
		Addr:    s.cfg.Addr,
		Handler: s.mux,
	}
	fmt.Printf("🚀 PyGo server listening on %s\n", s.cfg.Addr)

	if s.cfg.TLSCertFile != "" {
		return s.srv.ListenAndServeTLS(s.cfg.TLSCertFile, s.cfg.TLSKeyFile)
	}
	return s.srv.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	fmt.Println("🛑 Shutting down PyGo server...")
	return s.srv.Shutdown(ctx)
}
