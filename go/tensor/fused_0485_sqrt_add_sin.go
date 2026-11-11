package tensor

import "fmt"

func Fused0485SqrtAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Sin()
	return r, nil
}
func Fused0485SqrtAddSinName() string { return "sqrt_add_sin" }
