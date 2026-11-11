package tensor

import "fmt"

func Fused0278TanhDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Sigmoid()
	return r, nil
}
func Fused0278TanhDivSigmoidName() string { return "tanh_div_sigmoid" }
