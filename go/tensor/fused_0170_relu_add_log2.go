package tensor

import "fmt"

func Fused0170ReluAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Log2()
	return r, nil
}
func Fused0170ReluAddLog2Name() string { return "relu_add_log2" }
