package serialization

import (
	"github.com/pygo/go/tensor"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSafeTensorsRoundTripFloat32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roundtrip.safetensors")
	src := tensor.FromSlice([]float32{1.25, -2.5, 3.75, 4.5}, tensor.Shape{2, 2}, tensor.WithDType(tensor.Float32))
	w := NewSTWriter()
	w.Add("weights", src)
	if err := w.Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	r, err := LoadST(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.Get("weights")
	if err != nil {
		t.Fatal(err)
	}
	vals := got.Float32Slice()
	want := []float32{1.25, -2.5, 3.75, 4.5}
	if len(vals) != len(want) {
		t.Fatalf("got %d values, want %d", len(vals), len(want))
	}
	for i := range want {
		if math.Abs(float64(vals[i]-want[i])) > 1e-6 {
			t.Fatalf("value %d: got %v want %v", i, vals[i], want[i])
		}
	}
}
