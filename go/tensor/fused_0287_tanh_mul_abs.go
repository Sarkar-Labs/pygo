package tensor

import "fmt"

func Fused0287TanhMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Abs()
	return r, nil
}
func Fused0287TanhMulAbsName() string { return "tanh_mul_abs" }
