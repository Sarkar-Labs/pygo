package tensor

import "fmt"

func Fused1261GeluSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.ReLU()
	return r, nil
}
func Fused1261GeluSubReluName() string { return "gelu_sub_relu" }
