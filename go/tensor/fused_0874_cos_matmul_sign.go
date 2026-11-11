package tensor

import "fmt"

func Fused0874CosMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Sign()
	return r, nil
}
func Fused0874CosMatmulSignName() string { return "cos_matmul_sign" }
