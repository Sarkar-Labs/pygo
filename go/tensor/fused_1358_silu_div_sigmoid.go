package tensor

import "fmt"

func Fused1358SiluDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1358SiluDivSigmoidName() string { return "silu_div_sigmoid" }
