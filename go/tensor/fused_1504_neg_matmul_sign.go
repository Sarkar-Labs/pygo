package tensor

import "fmt"

func Fused1504NegMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Sign()
	return r, nil
}
func Fused1504NegMatmulSignName() string { return "neg_matmul_sign" }
