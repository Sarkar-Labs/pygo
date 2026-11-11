package tensor

import "fmt"

func Fused1566Log2SubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Sin()
	return r, nil
}
func Fused1566Log2SubSinName() string { return "log2_sub_sin" }
