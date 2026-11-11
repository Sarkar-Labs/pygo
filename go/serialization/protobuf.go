package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedProtobufEnvelope is a serializable record used by protobuf.
type GeneratedProtobufEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedProtobuf(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedProtobufEnvelope{Name: "protobuf", Version: 1, Data: data})
}
func DecodeGeneratedProtobuf(raw []byte) ([]byte, error) {
	var e GeneratedProtobufEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("protobuf: %w", err)
	}
	return e.Data, nil
}
