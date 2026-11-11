package tensor

import "fmt"

func Fused0780SinAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Sign()
	return r, nil
}
func Fused0780SinAddSignName() string { return "sin_add_sign" }
