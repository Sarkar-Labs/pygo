package tensor

import "fmt"

func Fused0756SinSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Sin()
	return r, nil
}
func Fused0756SinSubSinName() string { return "sin_sub_sin" }
