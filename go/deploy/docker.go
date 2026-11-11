package deploy

import (
	"fmt"
	"os"
	"text/template"
)

type DeployConfig struct {
	ModelName, ImageName, ImageTag string
	Port, Replicas                 int
	CPUReq, MemReq                 string
	GPUs                           int
	Namespace                      string
}

func Default() DeployConfig {
	return DeployConfig{"pygo-model", "pygo/serve", "latest", 8080, 2, "2", "4Gi", 1, "default"}
}

const dfTmpl = `FROM nvidia/cuda:12.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip golang-go
WORKDIR /app
COPY . .
RUN pip3 install torch
EXPOSE {{.Port}}
CMD ["./server"]
`

func WriteDockerfile(cfg DeployConfig, out string) error {
	t, err := template.New("df").Parse(dfTmpl)
	if err != nil {
		return err
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return t.Execute(f, cfg)
}

func WriteK8s(cfg DeployConfig, out string) error {
	content := fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: %s
  namespace: %s
spec:
  replicas: %d
`, cfg.ModelName, cfg.Namespace, cfg.Replicas)
	return os.WriteFile(out, []byte(content), 0644)
}
