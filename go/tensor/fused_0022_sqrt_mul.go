package tensor

import "fmt"

func Fused0022SqrtMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	return r, nil
}
func Fused0022SqrtMulName() string { return "sqrt_mul" }
