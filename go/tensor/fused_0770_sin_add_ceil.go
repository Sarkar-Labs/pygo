package tensor

import "fmt"

func Fused0770SinAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Ceil()
	return r, nil
}
func Fused0770SinAddCeilName() string { return "sin_add_ceil" }
