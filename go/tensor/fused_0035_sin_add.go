package tensor

import "fmt"

func Fused0035SinAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	return r, nil
}
func Fused0035SinAddName() string { return "sin_add" }
