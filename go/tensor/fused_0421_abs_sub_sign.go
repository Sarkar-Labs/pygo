package tensor

import "fmt"

func Fused0421AbsSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Sign()
	return r, nil
}
func Fused0421AbsSubSignName() string { return "abs_sub_sign" }
