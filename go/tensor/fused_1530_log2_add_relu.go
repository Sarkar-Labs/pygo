package tensor

import "fmt"

func Fused1530Log2AddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.ReLU()
	return r, nil
}
func Fused1530Log2AddReluName() string { return "log2_add_relu" }
