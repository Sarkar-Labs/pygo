package cgo

import "testing"

func TestBridgeTensorZeroCopy(t *testing.T) {
	bt, err := NewBridgeTensor([]int64{2, 2}, Float32, 0)
	if err != nil {
		t.Fatal(err)
	}
	data := bt.Float32Slice()
	if len(data) != 4 {
		t.Fatalf("got %d elements", len(data))
	}
	data[1] = 42
	if bt.Float32Slice()[1] != 42 {
		t.Fatal("Float32Slice is not backed by tensor storage")
	}
}

func TestBridgeTensorRejectsNegativeShape(t *testing.T) {
	if _, err := NewBridgeTensor([]int64{2, -1}, Float32, 0); err == nil {
		t.Fatal("expected negative shape dimension to be rejected")
	}
}

func TestBridgeTensorZeroDimMetadata(t *testing.T) {
	bt, err := NewBridgeTensor(nil, Float32, 0)
	if err != nil {
		t.Fatal(err)
	}
	if bt.NumEl() != 1 {
		t.Fatalf("got numel %d, want 1", bt.NumEl())
	}
}

func TestEmbeddedCPythonLifecycle(t *testing.T) {
	if err := InitPython(); err != nil {
		t.Fatal(err)
	}
	if !IsInitialized() {
		t.Fatal("CPython reports uninitialized after InitPython")
	}
	if err := RunPython("_pygo_bridge_probe = 579 + 3468"); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddedCPythonPreservesExceptionMessage(t *testing.T) {
	if err := InitPython(); err != nil {
		t.Fatal(err)
	}
	err := RunPython("raise RuntimeError('bridge-error-579')")
	if err == nil {
		t.Fatal("expected Python exception")
	}
	if got := err.Error(); got != "Python error: bridge-error-579" {
		t.Fatalf("got %q, want preserved Python exception", got)
	}
}
