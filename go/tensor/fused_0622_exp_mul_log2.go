package tensor

import "fmt"

func Fused0622ExpMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Log2()
	return r, nil
}
func Fused0622ExpMulLog2Name() string { return "exp_mul_log2" }
