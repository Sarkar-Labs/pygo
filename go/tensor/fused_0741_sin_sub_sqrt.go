package tensor

import "fmt"

func Fused0741SinSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Sqrt()
	return r, nil
}
func Fused0741SinSubSqrtName() string { return "sin_sub_sqrt" }
