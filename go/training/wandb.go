package training

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

type WandbConfig struct{ Project, Name, Entity, APIKey string }
type WandbLogger struct {
	mu      sync.Mutex
	cfg     WandbConfig
	runID   string
	step    int
	client  *http.Client
	enabled bool
}

func NewWandbLogger(cfg WandbConfig) *WandbLogger {
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv("WANDB_API_KEY")
	}
	return &WandbLogger{cfg: cfg, client: &http.Client{Timeout: 5 * time.Second}, enabled: key != ""}
}
func (w *WandbLogger) Init() {
	if !w.enabled {
		fmt.Println("WandB: disabled")
		return
	}
	w.runID = fmt.Sprintf("run_%d", time.Now().Unix())
	fmt.Printf("WandB: %s/%s/%s\n", w.cfg.Entity, w.cfg.Project, w.runID)
}
func (w *WandbLogger) Log(metrics map[string]interface{}) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.step++
	if !w.enabled {
		return
	}
	payload := map[string]interface{}{"step": w.step}
	for k, v := range metrics {
		payload[k] = v
	}
	body, _ := json.Marshal(payload)
	go func() {
		req, _ := http.NewRequest("POST", "https://api.wandb.ai/graphql", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		if resp, err := w.client.Do(req); err == nil {
			resp.Body.Close()
		}
	}()
}
func (w *WandbLogger) Finish() {
	if w.enabled {
		fmt.Printf("WandB: finished %s\n", w.runID)
	}
}
