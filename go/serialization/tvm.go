package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedTvmEnvelope is a serializable record used by tvm.
type GeneratedTvmEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedTvm(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedTvmEnvelope{Name: "tvm", Version: 1, Data: data})
}
func DecodeGeneratedTvm(raw []byte) ([]byte, error) {
	var e GeneratedTvmEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("tvm: %w", err)
	}
	return e.Data, nil
}
