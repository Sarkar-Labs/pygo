package tensor

import "fmt"

func Fused0036SinSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	return r, nil
}
func Fused0036SinSubName() string { return "sin_sub" }
