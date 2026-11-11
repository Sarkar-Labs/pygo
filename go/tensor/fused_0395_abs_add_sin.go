package tensor

import "fmt"

func Fused0395AbsAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Sin()
	return r, nil
}
func Fused0395AbsAddSinName() string { return "abs_add_sin" }
