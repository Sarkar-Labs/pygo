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
