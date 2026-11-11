package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/pygo/go/tensor"
	"net/http"
	"time"
)

type TritonConfig struct {
	ServerURL, ModelName, ModelVersion string
	Timeout                            time.Duration
	MaxRetries                         int
}

func DefaultTritonConfig(url, model string) TritonConfig {
	return TritonConfig{ServerURL: url, ModelName: model, ModelVersion: "1", Timeout: 30 * time.Second, MaxRetries: 3}
}

type TritonClient struct {
	cfg    TritonConfig
	client *http.Client
}

func NewTritonClient(cfg TritonConfig) *TritonClient {
	return &TritonClient{cfg: cfg, client: &http.Client{Timeout: cfg.Timeout}}
}
func (c *TritonClient) Infer(ctx context.Context, inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	body, _ := json.Marshal(map[string]interface{}{"n": len(inputs)})
	url := fmt.Sprintf("%s/v2/models/%s/versions/%s/infer", c.cfg.ServerURL, c.cfg.ModelName, c.cfg.ModelVersion)
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return make(map[string]*tensor.Tensor), nil
}
func (c *TritonClient) Health(ctx context.Context) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.cfg.ServerURL+"/v2/health/ready", nil)
	resp, err := c.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200, nil
}
