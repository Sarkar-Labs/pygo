package tensor

import "fmt"

func Fused0514SqrtMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Sign()
	return r, nil
}
func Fused0514SqrtMatmulSignName() string { return "sqrt_matmul_sign" }
