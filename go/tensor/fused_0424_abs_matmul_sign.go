package tensor

import "fmt"

func Fused0424AbsMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Sign()
	return r, nil
}
func Fused0424AbsMatmulSignName() string { return "abs_matmul_sign" }
