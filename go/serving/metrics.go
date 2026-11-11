package serving

import (
	"fmt"
	"sync/atomic"
	"time"
)

type ServerMetrics struct {
	RequestCount      atomic.Int64
	ErrorCount        atomic.Int64
	TotalLatencyMS    atomic.Int64
	ActiveConnections atomic.Int32
	BatchCount        atomic.Int64
	TokensGenerated   atomic.Int64
	StartTime         time.Time
}

func NewMetrics() *ServerMetrics { return &ServerMetrics{StartTime: time.Now()} }
func (m *ServerMetrics) RecordRequest(latencyMS float64, isError bool) {
	m.RequestCount.Add(1)
	m.TotalLatencyMS.Add(int64(latencyMS))
	if isError {
		m.ErrorCount.Add(1)
	}
}
func (m *ServerMetrics) Record(latencyMS float64, isErr bool, tokens int64) {
	m.RecordRequest(latencyMS, isErr)
	m.TokensGenerated.Add(tokens)
}
func (m *ServerMetrics) Snapshot() map[string]interface{} {
	n := m.RequestCount.Load()
	avg := 0.0
	if n > 0 {
		avg = float64(m.TotalLatencyMS.Load()) / float64(n)
	}
	up := time.Since(m.StartTime).Seconds()
	rps := 0.0
	if up > 0 {
		rps = float64(n) / up
	}
	return map[string]interface{}{"requests_total": n, "requests_error": m.ErrorCount.Load(), "avg_latency_ms": avg, "tokens_generated": m.TokensGenerated.Load(), "active_connections": m.ActiveConnections.Load(), "batch_count": m.BatchCount.Load(), "uptime_seconds": up, "requests_per_sec": rps}
}
func (m *ServerMetrics) Prometheus() string {
	out := ""
	for k, v := range m.Snapshot() {
		out += fmt.Sprintf("pygo_%s %v\n", k, v)
	}
	return out
}
