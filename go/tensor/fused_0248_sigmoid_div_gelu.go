package tensor

import "fmt"

func Fused0248SigmoidDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.GELU()
	return r, nil
}
func Fused0248SigmoidDivGeluName() string { return "sigmoid_div_gelu" }
