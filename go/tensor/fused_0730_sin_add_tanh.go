package tensor

import "fmt"

func Fused0730SinAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_tanh: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Tanh()
	return r, nil
}
func Fused0730SinAddTanhName() string { return "sin_add_tanh" }
