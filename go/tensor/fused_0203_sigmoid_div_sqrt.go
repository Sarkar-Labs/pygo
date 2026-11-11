package tensor

import "fmt"

func Fused0203SigmoidDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Sqrt()
	return r, nil
}
func Fused0203SigmoidDivSqrtName() string { return "sigmoid_div_sqrt" }
