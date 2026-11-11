package tensor

import "fmt"

func Fused0511SqrtSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Sign()
	return r, nil
}
func Fused0511SqrtSubSignName() string { return "sqrt_sub_sign" }
