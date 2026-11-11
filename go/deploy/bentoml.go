package deploy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

type BentoMLConfig struct{ ServerURL, BentoName, ServiceName, APIToken string }

func DefaultBentoMLConfig(name string) BentoMLConfig {
	return BentoMLConfig{"http://localhost:3000", name, "predict", os.Getenv("BENTO_API_KEY")}
}

type BentoMLClient struct {
	cfg    BentoMLConfig
	client *http.Client
}

func NewBentoMLClient(cfg BentoMLConfig) *BentoMLClient {
	return &BentoMLClient{cfg: cfg, client: &http.Client{Timeout: 30 * time.Second}}
}
func (b *BentoMLClient) Predict(input interface{}) (map[string]interface{}, error) {
	body, _ := json.Marshal(input)
	req, _ := http.NewRequest("POST", fmt.Sprintf("%s/%s", b.cfg.ServerURL, b.cfg.ServiceName), bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	if b.cfg.APIToken != "" {
		req.Header.Set("Authorization", "Bearer "+b.cfg.APIToken)
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	return result, nil
}
func (b *BentoMLClient) Health() (bool, error) {
	resp, err := b.client.Get(b.cfg.ServerURL + "/healthz")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200, nil
}
