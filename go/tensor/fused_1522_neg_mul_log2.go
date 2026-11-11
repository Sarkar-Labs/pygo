package tensor

import "fmt"

func Fused1522NegMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Log2()
	return r, nil
}
func Fused1522NegMulLog2Name() string { return "neg_mul_log2" }
