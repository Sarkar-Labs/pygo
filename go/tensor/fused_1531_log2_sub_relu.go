package tensor

import "fmt"

func Fused1531Log2SubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.ReLU()
	return r, nil
}
func Fused1531Log2SubReluName() string { return "log2_sub_relu" }
