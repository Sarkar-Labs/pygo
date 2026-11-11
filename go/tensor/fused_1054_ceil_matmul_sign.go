package tensor

import "fmt"

func Fused1054CeilMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Sign()
	return r, nil
}
func Fused1054CeilMatmulSignName() string { return "ceil_matmul_sign" }
