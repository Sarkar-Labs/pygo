package tensor

import "fmt"

func Fused0731SinSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Tanh()
	return r, nil
}
func Fused0731SinSubTanhName() string { return "sin_sub_tanh" }
