package tensor

import "fmt"

func Fused0081Log2Sub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	return r, nil
}
func Fused0081Log2SubName() string { return "log2_sub" }
