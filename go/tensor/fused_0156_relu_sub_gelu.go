package tensor

import "fmt"

func Fused0156ReluSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.GELU()
	return r, nil
}
func Fused0156ReluSubGeluName() string { return "relu_sub_gelu" }
