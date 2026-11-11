package tensor

import "fmt"

func Fused0246SigmoidSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.GELU()
	return r, nil
}
func Fused0246SigmoidSubGeluName() string { return "sigmoid_sub_gelu" }
