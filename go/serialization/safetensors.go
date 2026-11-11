package serialization

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/pygo/go/tensor"
	"math"
	"os"
)

type TensorMeta struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type SafeTensorsWriter struct {
	tensors  map[string]*tensor.Tensor
	metadata map[string]string
}

func dtStr(dt tensor.DType) string {
	switch dt {
	case tensor.Float32:
		return "F32"
	case tensor.Float16:
		return "F16"
	case tensor.BFloat16:
		return "BF16"
	case tensor.Float64:
		return "F64"
	case tensor.Int8:
		return "I8"
	case tensor.Int16:
		return "I16"
	case tensor.Int32:
		return "I32"
	case tensor.Int64:
		return "I64"
	case tensor.UInt8:
		return "U8"
	default:
		return "UNKNOWN"
	}
}

func NewSTWriter() *SafeTensorsWriter {
	return &SafeTensorsWriter{tensors: make(map[string]*tensor.Tensor), metadata: make(map[string]string)}
}
func (w *SafeTensorsWriter) Add(name string, t *tensor.Tensor) { w.tensors[name] = t }
func (w *SafeTensorsWriter) Meta(k, v string)                  { w.metadata[k] = v }
func (w *SafeTensorsWriter) Save(path string) error {
	type entry struct {
		meta TensorMeta
		data []byte
	}
	entries := make(map[string]entry, len(w.tensors))
	offset := int64(0)
	for name, t := range w.tensors {
		if t == nil {
			return fmt.Errorf("nil tensor %q", name)
		}
		if t.DType() != tensor.Float32 {
			return fmt.Errorf("safetensors writer currently supports float32, %q is %s", name, t.DType())
		}
		vals := t.Float32Slice()
		data := make([]byte, len(vals)*4)
		for i, v := range vals {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
		entries[name] = entry{TensorMeta{dtStr(t.DType()), []int64(t.Shape()), [2]int64{offset, offset + int64(len(data))}}, data}
		offset += int64(len(data))
	}
	header := make(map[string]interface{}, len(entries)+1)
	if len(w.metadata) > 0 {
		header["__metadata__"] = w.metadata
	}
	for name, e := range entries {
		header[name] = e.meta
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var hs [8]byte
	binary.LittleEndian.PutUint64(hs[:], uint64(len(hb)))
	if _, err = f.Write(hs[:]); err != nil {
		return err
	}
	if _, err = f.Write(hb); err != nil {
		return err
	}
	for _, e := range entries {
		if _, err = f.Write(e.data); err != nil {
			return err
		}
	}
	return nil
}

type SafeTensorsReader struct {
	path   string
	header map[string]TensorMeta
}

func LoadST(path string) (*SafeTensorsReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hs uint64
	binary.Read(f, binary.LittleEndian, &hs)
	hb := make([]byte, hs)
	f.Read(hb)
	var raw map[string]json.RawMessage
	json.Unmarshal(hb, &raw)
	r := &SafeTensorsReader{path: path, header: make(map[string]TensorMeta)}
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var m TensorMeta
		if json.Unmarshal(v, &m) == nil {
			r.header[k] = m
		}
	}
	return r, nil
}
func (r *SafeTensorsReader) Keys() []string {
	ks := make([]string, 0, len(r.header))
	for k := range r.header {
		ks = append(ks, k)
	}
	return ks
}
func (r *SafeTensorsReader) Get(name string) (*tensor.Tensor, error) {
	m, ok := r.header[name]
	if !ok {
		return nil, fmt.Errorf("not found: %s", name)
	}
	if m.DType != "F32" {
		return nil, fmt.Errorf("unsupported dtype %s", m.DType)
	}
	f, err := os.Open(r.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hs uint64
	if err := binary.Read(f, binary.LittleEndian, &hs); err != nil {
		return nil, err
	}
	if hs > 64<<20 {
		return nil, fmt.Errorf("header too large: %d", hs)
	}
	if _, err := f.Seek(int64(8+hs)+m.DataOffsets[0], 0); err != nil {
		return nil, err
	}
	n := int64(1)
	for _, d := range m.Shape {
		n *= d
	}
	data := make([]float32, n)
	raw := make([]byte, n*4)
	if _, err := f.Read(raw); err != nil {
		return nil, err
	}
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return tensor.FromSlice(data, tensor.Shape(m.Shape), tensor.WithDType(tensor.Float32)), nil
}
