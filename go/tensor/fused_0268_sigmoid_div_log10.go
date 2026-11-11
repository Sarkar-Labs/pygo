package tensor

import "fmt"

func Fused0268SigmoidDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Log10()
	return r, nil
}
func Fused0268SigmoidDivLog10Name() string { return "sigmoid_div_log10" }
