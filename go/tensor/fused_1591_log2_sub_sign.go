package tensor

import "fmt"

func Fused1591Log2SubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Sign()
	return r, nil
}
func Fused1591Log2SubSignName() string { return "log2_sub_sign" }
