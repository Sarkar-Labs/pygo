package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedMsgpackEnvelope is a serializable record used by msgpack.
type GeneratedMsgpackEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedMsgpack(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedMsgpackEnvelope{Name: "msgpack", Version: 1, Data: data})
}
func DecodeGeneratedMsgpack(raw []byte) ([]byte, error) {
	var e GeneratedMsgpackEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("msgpack: %w", err)
	}
	return e.Data, nil
}
