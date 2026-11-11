package tensor

import "fmt"

func Fused0740SinAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Sqrt()
	return r, nil
}
func Fused0740SinAddSqrtName() string { return "sin_add_sqrt" }
