package tensor

import "fmt"

func Fused1592Log2MulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Sign()
	return r, nil
}
func Fused1592Log2MulSignName() string { return "log2_mul_sign" }
