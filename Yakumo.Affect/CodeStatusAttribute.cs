/// <summary>カスタムタグ属性: コードの状態管理用</summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method | AttributeTargets.Property)]
public sealed class CodeStatusAttribute : Attribute
{
    public CodeStatus Status { get; }
    public string? Note { get; init; }
    public CodeStatusAttribute(CodeStatus status) => Status = status;
}

public enum CodeStatus
{
    /// <summary>現役コード</summary>
    Active,
    /// <summary>研究・フォールバック用として保持（GoEmotions移行後の旧パス）</summary>
    Legacy,
    /// <summary>実験的・評価中</summary>
    Experimental,
    /// <summary>将来削除予定</summary>
    Deprecated,
}