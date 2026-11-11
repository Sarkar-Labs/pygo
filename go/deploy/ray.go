package deploy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type RayServeConfig struct {
	RayAddress, DeploymentName string
	NumReplicas                int
	NumCPUs, NumGPUs           float64
	MaxConcurrency             int
}

func DefaultRayConfig(name string) RayServeConfig {
	return RayServeConfig{"http://localhost:8265", name, 2, 4, 1, 32}
}

type RayServeClient struct {
	cfg    RayServeConfig
	client *http.Client
}

func NewRayServeClient(cfg RayServeConfig) *RayServeClient {
	return &RayServeClient{cfg: cfg, client: &http.Client{Timeout: 30 * time.Second}}
}
func (r *RayServeClient) Deploy() error {
	body, _ := json.Marshal(map[string]interface{}{"name": r.cfg.DeploymentName, "num_replicas": r.cfg.NumReplicas, "ray_actor_options": map[string]interface{}{"num_cpus": r.cfg.NumCPUs, "num_gpus": r.cfg.NumGPUs}, "max_concurrent_queries": r.cfg.MaxConcurrency})
	req, _ := http.NewRequest("PUT", r.cfg.RayAddress+"/api/serve/deployments/", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	fmt.Printf("Ray: deployed %s\n", r.cfg.DeploymentName)
	return nil
}
func (r *RayServeClient) Scale(n int) error {
	body, _ := json.Marshal(map[string]interface{}{"name": r.cfg.DeploymentName, "num_replicas": n})
	req, _ := http.NewRequest("PATCH", fmt.Sprintf("%s/api/serve/deployments/%s", r.cfg.RayAddress, r.cfg.DeploymentName), bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	fmt.Printf("Ray: scaled -> %d\n", n)
	return nil
}
