package tensor

import "fmt"

func Fused1320GeluAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Sign()
	return r, nil
}
func Fused1320GeluAddSignName() string { return "gelu_add_sign" }
