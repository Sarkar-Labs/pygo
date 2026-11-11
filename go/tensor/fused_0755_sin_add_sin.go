package tensor

import "fmt"

func Fused0755SinAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Sin()
	return r, nil
}
func Fused0755SinAddSinName() string { return "sin_add_sin" }
