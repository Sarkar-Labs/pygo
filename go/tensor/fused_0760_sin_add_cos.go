package tensor

import "fmt"

func Fused0760SinAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Cos()
	return r, nil
}
func Fused0760SinAddCosName() string { return "sin_add_cos" }
