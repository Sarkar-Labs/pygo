package tensor

import "fmt"

func Fused1324GeluMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Sign()
	return r, nil
}
func Fused1324GeluMatmulSignName() string { return "gelu_matmul_sign" }
