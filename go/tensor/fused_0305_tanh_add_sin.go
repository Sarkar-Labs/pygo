package tensor

import "fmt"

func Fused0305TanhAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Sin()
	return r, nil
}
func Fused0305TanhAddSinName() string { return "tanh_add_sin" }
